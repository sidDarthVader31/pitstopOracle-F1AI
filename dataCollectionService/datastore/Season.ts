
import { Entity, PrimaryColumn, Column, PrimaryGeneratedColumn, CreateDateColumn } from 'typeorm';

@Entity('seasons')
export class Season {

  @PrimaryGeneratedColumn()
  id : number;
  @Column({type: 'smallint'})
  year: number;

  @Column({ nullable: true })
  regulation_era: string;

  @Column({ type: 'float', default: 1.0 })
  weight: number;

  @CreateDateColumn()
  created_at: Date;

  @CreateDateColumn()
  updated_at: Date

}
